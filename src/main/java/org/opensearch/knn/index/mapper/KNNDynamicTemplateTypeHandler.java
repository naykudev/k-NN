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
import java.util.Map;

/**
 * k-NN implementation of {@link DynamicTemplateTypeHandler}.
 *
 * <p>Registered against a dynamic template with {@code match_mapping_type: "array"}. Core detects only
 * that an unmapped field's value is an array and offers the matched template to this handler, which
 * decides whether the array is actually a {@code knn_vector} and, if so, completes the config before
 * {@code KNNVectorFieldMapper.TypeParser} builds the mapper.
 *
 * <p>The claim decision depends on the template's {@code mapping} block:
 * <ul>
 *   <li><b>Explicit knn intent</b> — the block sets {@code type: knn_vector}, or supplies a knn-specific
 *       parameter ({@code dimension} or {@code model_id}) with no other type. The field is claimed with
 *       no length threshold; {@code dimension} is injected from the array length only when neither it nor
 *       a {@code model_id} is present.</li>
 *   <li><b>Empty block</b> — no type and no knn signal. The array length decides: {@code >= }
 *       {@link #MIN_VECTOR_DIMENSION} is claimed as {@code knn_vector} (dimension injected), otherwise the
 *       handler declines and the field falls through to normal element-wise array parsing.</li>
 *   <li><b>A different explicit type</b> — declined; not ours.</li>
 * </ul>
 *
 * <p>A fully-specified config ({@code dimension} or {@code model_id} present) never opens a parser, so
 * core can also validate such templates eagerly at index-creation time.
 */
public class KNNDynamicTemplateTypeHandler implements DynamicTemplateTypeHandler {

    static final int MIN_VECTOR_DIMENSION = 128;

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
        // An explicit type that isn't knn_vector is not ours — decline and let another handler or the
        // normal array path take it.
        if (typeNode != null && KNNVectorFieldMapper.CONTENT_TYPE.equals(typeNode.toString()) == false) {
            return false;
        }

        // Explicit knn intent: the user wrote type: knn_vector, or gave a knn-specific parameter
        // (dimension / model_id). Honor it with no length threshold.
        boolean explicitKnn = typeNode != null || isConfigComplete(mappingConfig);
        if (explicitKnn) {
            mappingConfig.putIfAbsent("type", KNNVectorFieldMapper.CONTENT_TYPE);
            // A complete config needs no data-derived parameter, so we must not open the parser: doing so
            // would defer index-creation-time validation, and injecting a data-derived dimension alongside
            // a model_id is rejected by the TypeParser.
            if (isConfigComplete(mappingConfig)) {
                return true;
            }
            // type: knn_vector without a dimension — inject it from the array length (no threshold). If the
            // value is not an array, leave the config as-is; the TypeParser reports the missing dimension.
            try (XContentParser parser = fieldValueParser.get()) {
                if (parser.currentToken() == XContentParser.Token.START_ARRAY) {
                    mappingConfig.put(KNNConstants.DIMENSION, countArray(parser));
                }
            }
            return true;
        }

        // Empty block, no explicit type or knn signal: let the array length decide. Only claim arrays at
        // or above the minimum dimension; anything shorter (or non-array) falls through to normal parsing.
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
