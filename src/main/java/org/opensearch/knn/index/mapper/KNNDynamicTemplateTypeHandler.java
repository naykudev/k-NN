/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.DynamicTemplateTypeHandler;
import org.opensearch.index.mapper.FieldValueParserSupplier;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * k-NN implementation of {@link DynamicTemplateTypeHandler}.
 *
 * <p>Registered against a dynamic template with {@code match_mapping_type: "array"}. Core detects only
 * that an unmapped field's value is an array and offers the matched template to this handler, which
 * decides whether the array is actually a {@code knn_vector} and, if so, completes the config before
 * {@code KNNVectorFieldMapper.TypeParser} builds the mapper.
 *
 * <p>The decision is driven by what the template's {@code mapping} block contains:
 * <ul>
 *   <li><b>A foreign type</b> ({@code type} set to something other than {@code knn_vector}) — decline;
 *       not ours.</li>
 *   <li><b>An explicit {@code type: knn_vector}</b> — claim. Any unknown parameter is left in the config
 *       so {@code KNNVectorFieldMapper.TypeParser} reports it with a clear error, exactly as a
 *       misconfigured explicit mapping would. {@code dimension} is inferred from the array length when
 *       absent (no threshold).</li>
 *   <li><b>A knn-specific parameter</b> ({@code dimension}, {@code model_id}, {@code data_type},
 *       {@code method}, {@code engine}, {@code space_type}, {@code mode}, {@code compression_level}) with
 *       no type — strong intent, so claim with no threshold and inject the type; infer {@code dimension}
 *       when neither it nor a {@code model_id} is present.</li>
 *   <li><b>An unknown parameter with no knn type/signal</b> — decline; this template isn't ours, so it
 *       falls through to the next handler or normal array parsing rather than hard-failing.</li>
 *   <li><b>Only generic parameters ({@code store}, {@code doc_values}, {@code meta}) or an empty block</b>
 *       — intent-neutral, so the array length decides: {@code >= }{@link #MIN_VECTOR_DIMENSION} is claimed
 *       (type + dimension injected), otherwise declined.</li>
 * </ul>
 *
 * <p>A fully-specified config ({@code dimension} or {@code model_id} present) never opens a parser, so
 * core can also validate such templates eagerly at index-creation time.
 */
public class KNNDynamicTemplateTypeHandler implements DynamicTemplateTypeHandler {

    static final int MIN_VECTOR_DIMENSION = 128;

    /**
     * knn-specific mapping parameters. Any of these present in the block signals the user intends a
     * knn_vector, so the field is claimed without applying the length threshold.
     */
    private static final Set<String> KNN_SPECIFIC_PARAMS = Set.of(
        KNNConstants.DIMENSION,
        KNNConstants.MODEL_ID,
        KNNConstants.VECTOR_DATA_TYPE_FIELD,
        KNNConstants.KNN_METHOD,
        KNNConstants.KNN_ENGINE,
        KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE,
        KNNConstants.MODE_PARAMETER,
        KNNConstants.COMPRESSION_LEVEL_PARAMETER
    );

    /**
     * Generic field-mapping parameters that every field type accepts. They carry no vector intent, so a
     * block containing only these is treated like an empty block (length threshold decides).
     */
    private static final Set<String> GENERIC_PARAMS = Set.of("store", "doc_values", "meta", "index", "boost");

    /** The full set of keys a knn_vector mapping config may contain (type plus all valid params). */
    private static final Set<String> KNOWN_KEYS;
    static {
        KNOWN_KEYS = new HashSet<>();
        KNOWN_KEYS.add("type");
        KNOWN_KEYS.addAll(KNN_SPECIFIC_PARAMS);
        KNOWN_KEYS.addAll(GENERIC_PARAMS);
    }

    /**
     * Decides whether the matched {@code array} template's field is a {@code knn_vector} and, if so,
     * completes the mapping config (injects {@code type} and, when needed, {@code dimension}).
     *
     * @param mappingConfig the mutable mapping config from the matched template
     * @param fieldValueParser produces a fresh parser positioned at the field value's first token
     * @return {@code true} if this handler claims the field as a knn_vector, {@code false} to decline it
     */
    @Override
    public boolean adjustMappingConfig(Map<String, Object> mappingConfig, FieldValueParserSupplier fieldValueParser) throws IOException {
        Object typeNode = mappingConfig.get("type");
        boolean explicitKnnType = typeNode != null && KNNVectorFieldMapper.CONTENT_TYPE.equals(typeNode.toString());

        // A foreign explicit type is not ours — decline and let another handler or the normal array path take it.
        if (typeNode != null && explicitKnnType == false) {
            return false;
        }

        // With an explicit type: knn_vector the user has declared intent. Claim it and leave any unknown
        // parameter in place so the TypeParser reports a clear error (e.g. a "dimesnion" typo), rather than
        // silently declining into a float array.
        if (explicitKnnType) {
            return claimAndInferDimension(mappingConfig, fieldValueParser);
        }

        // No type given. Inspect the remaining keys to gauge intent.
        boolean hasKnnParam = false;
        boolean hasUnknownParam = false;
        for (String key : mappingConfig.keySet()) {
            if (KNN_SPECIFIC_PARAMS.contains(key)) {
                hasKnnParam = true;
            } else if (KNOWN_KEYS.contains(key) == false) {
                hasUnknownParam = true;
            }
        }

        // A knn-specific parameter is a strong signal even without a type — claim with no threshold.
        if (hasKnnParam) {
            return claimAndInferDimension(mappingConfig, fieldValueParser);
        }

        // An unknown parameter with no knn type or signal means this template isn't ours — decline so it
        // falls through to the next handler or normal array parsing.
        if (hasUnknownParam) {
            return false;
        }

        // Only generic params (store/doc_values/meta) or an empty block: intent-neutral, so the array
        // length decides. Claim only arrays at or above the minimum dimension; shorter or non-array declines.
        try (XContentParser parser = fieldValueParser.get()) {
            if (parser.currentToken() != XContentParser.Token.START_ARRAY) {
                return false;
            }
            int count = countArray(parser);
            if (count < MIN_VECTOR_DIMENSION) {
                return false;
            }
            mappingConfig.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
            mappingConfig.put(KNNConstants.DIMENSION, count);
        }
        return true;
    }

    /**
     * Claims the field as a knn_vector: injects the type when absent, and — if the config is not already
     * complete — infers {@code dimension} from the array length with no threshold. A complete config
     * (dimension or model_id present) never opens the parser.
     */
    private boolean claimAndInferDimension(Map<String, Object> mappingConfig, FieldValueParserSupplier fieldValueParser)
        throws IOException {
        mappingConfig.putIfAbsent("type", KNNVectorFieldMapper.CONTENT_TYPE);
        // A complete config needs no data-derived parameter, so we must not open the parser: doing so would
        // defer index-creation-time validation, and injecting a data-derived dimension alongside a model_id
        // is rejected by the TypeParser.
        if (isConfigComplete(mappingConfig)) {
            return true;
        }
        // Infer dimension from the array length. If the value is not an array, leave the config as-is; the
        // TypeParser reports the missing dimension.
        try (XContentParser parser = fieldValueParser.get()) {
            if (parser.currentToken() == XContentParser.Token.START_ARRAY) {
                mappingConfig.put(KNNConstants.DIMENSION, countArray(parser));
            }
        }
        return true;
    }

    /** Counts the elements of the array the parser is currently positioned on (at START_ARRAY). */
    private static int countArray(XContentParser parser) throws IOException {
        int count = 0;
        while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
            count++;
        }
        return count;
    }

    /**
     * A knn_vector config is fully specified when the dimension is given directly, or when a
     * {@code model_id} supplies it. In both cases the mapper can be built without inspecting a
     * document, so core can validate the template eagerly at index-creation time.
     */
    @Override
    public boolean isConfigComplete(Map<String, Object> mappingConfig) {
        return mappingConfig.containsKey(KNNConstants.DIMENSION) || mappingConfig.containsKey(KNNConstants.MODEL_ID);
    }
}
